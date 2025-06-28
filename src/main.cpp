
#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

static inline std::vector<std::string> tokenize(const std::string &s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string w;
    while (ss >> w) {
        std::transform(w.begin(), w.end(), w.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
        out.push_back(w);
    }
    return out;
}

struct Sample {
    std::string sentence;
    int label; // 0: anger, 1: happy, 2: sad, 3:fear, 4: neutral
};

static const std::vector<Sample> TRAIN{
    {"i am so angry right now", 0},
    {"this makes me furious", 0},
    {"what a wonderful day", 1},
    {"i feel very happy", 1},
    {"i am sad and lonely", 2},
    {"this is depressing", 2},
    {"i am scared of the dark", 3},
    {"this terrifies me", 3},
};

struct Vocab {

    Vocab(int max_size) : max_size(max_size) {}

    std::unordered_map<std::string, int> id_of;
    std::vector<std::string> word_of;
    size_t max_size;

    int operator[](const std::string &word) {
        auto it = id_of.find(word);
        int result = -1;
        if (it != id_of.end()) {
            result = it->second;
        }
        return result;
    }
    int size() const { return static_cast<int>(word_of.size()); }

    void init(const std::vector<std::string> &data) {
        std::unordered_map<std::string, int> freq;
        for (auto &s : data) {
            freq[s]++;
        }

        // horrible
        // FIXME: there is a much better way to do that using heaps
        std::vector<std::pair<std::string, int>> pairs;
        for (auto elem : freq) {
            pairs.emplace_back(std::make_pair(elem.first, elem.second));
        }

        std::sort(pairs.begin(), pairs.end(), [](auto &a, auto &b) {
            return a.second > b.second;
        });

        for (int i = 0, size = std::min(max_size, pairs.size()); i < size; ++i) {
            auto & pair = pairs[i];
            this->word_of.push_back(pair.first);
            this->id_of[pair.first] = i;
        }

    }
};

static Vec bow(const std::string &sentence, Vocab &vocab) {
    /*for (const auto &tok : tokenize(sentence)) {
        vocab[tok]; // Add token to vocabulary
    }*/
    Vec v = Vec::Zero(vocab.size());
    for (const auto &tok : tokenize(sentence)) {
        int id = vocab[tok];
        if (id > -1) {
            v[id] += 1.0; // Increment the count for the token
        }
    }
    //std::cout << "Vocabulary size: " << vocab.size() << ", Vector size: " << v.size() << std::endl;
    return v;
}

struct Dense {
    Mat mat;
    Vec vec;

    Dense(int in, int out) {
        std::mt19937 gen {std::random_device{}()};
        std::normal_distribution<> nd{0.0, 1.0 / std::sqrt(in)};
        mat = Mat(out, in);
        for (int i = 0; i < mat.size(); ++i) mat.data()[i] = nd(gen);
        vec = Vec::Zero(out);
    }
    Vec forward(const Vec &x) const {
        //std::cout << "Forward pass with input vector of size: " << x.size() << " : " << mat.cols() << std::endl;
        assert(x.size() == mat.cols() && "Input vector size does not match matrix dimensions");
        return mat * x + vec;
    }
};

static Vec relu(const Vec &z) {
    return z.unaryExpr([](double v) { return v > 0 ? v : 0; });
}

static Vec relu_grad(const Vec &z) {
    return z.unaryExpr([](double v) { return v > 0 ? 1.0 : 0.0; });
}

static Vec softmax(const Vec &z) {
    double m = z.maxCoeff();
    Vec e = (z.array() - m).exp();
    return e / e.sum();
}

static double cross_entropy(const Vec &y, const Vec &t) {
    return - (t.array() * y.array().log()).sum();
}

struct Model {
    Dense h1;
    Dense out;
    double loss;

    Model(int input_dim, int hidden_dim, int output_dim, double loss = 0.05)
        : h1(input_dim, hidden_dim), out(hidden_dim, output_dim), loss(loss) {}

    double train_step(const Vec &x, int label) {
        Vec z1 = h1.forward(x);
        Vec a1 = relu(z1);
        Vec z2 = out.forward(a1);
        Vec y = softmax(z2);

        //target one hot
        Vec t = Vec::Zero(y.size());
        t(label) = 1.0;

        //loss
        double L = cross_entropy(y, t);


        //backward
        Vec dL_dz2 = y - t;
        Mat dL_dmat2 = dL_dz2 * a1.transpose();
        Vec dL_dvec2 = dL_dz2;

        Vec dL_da1 = out.mat.transpose() * dL_dz2;
        Vec dL_dz1 = dL_da1.cwiseProduct(relu_grad(z1));
        Mat dL_dmat1 = dL_dz1 * x.transpose();
        Vec dL_dvec1 = dL_dz1;

        out.mat -= loss * dL_dmat2;
        out.vec -= loss * dL_dvec2;
        h1.mat -= loss * dL_dmat1;
        h1.vec -= loss * dL_dvec1;

        return L;
    }

    int predict(const Vec &x) const {
        Vec a1 = relu(h1.forward(x));
        Vec y = softmax(out.forward(a1));
        Eigen::Index idx;
        y.maxCoeff(&idx);
        return static_cast<int>(idx);
    }
};


int main(){
    const int MAX_TOKENS = 1000;
    const int MAX_LENGTH = 250;
    Vocab vocab (MAX_TOKENS);

    // Build vocabulary
    std::vector<std::string> words;
    for (auto &instance : TRAIN) {
        auto sentence_words = tokenize(instance.sentence);
        words.insert(words.end(), sentence_words.begin(), sentence_words.end());
    }
    vocab.init(words);

    std::vector<Vec> inputs;
    std::vector<int> labels;

    const auto size = vocab.size();

    for (auto &instance : TRAIN) {
        Vec row = bow(instance.sentence, vocab);
        inputs.push_back(row);
        labels.push_back(instance.label);
    }

    const int input_dim = vocab.size();
    const int hidden_dim = 32;
    const int output_dim = 4;

    assert(input_dim > 0 && "Input dimension must be greater than 0");
    assert(inputs[0].size() == input_dim && "Input vector size must match input dimension");
    Model model(input_dim, hidden_dim, output_dim);

    //Training
    for (int epoch = 0; epoch < 1000; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            total_loss += model.train_step(inputs[i], labels[i]);
        }
        if (epoch % 100 == 0)
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / inputs.size() << std::endl;
    }

    // demo
    std::vector<std::string> test_sent{
        "i am absolutely terrified", // fear
        "this is awesome",           // happy
        "i hate everything",         // anger
        "i feel so empty"            // sad
    };

    const char *EMO[4] = {"anger", "happy", "sad", "fear"};

    for (const auto &s : test_sent) {
        Vec v = bow(s, vocab);
        int pred = model.predict(v);
        std::cout << std::left << std::setw(30) << ('"' + s + '"')
                  << " → " << EMO[pred] << '\n';
    }
    std::cout << "Finish\n";
    return 0;
}