#include <iostream>
#include "ProjetOption.h"
#include <stdexcept>
#include <cmath>

// class Option
Option::Option() {
    _expiry = 0;
}

Option::Option(double exp) {
    _expiry = exp;
}

double Option::getExpiry() {
    return _expiry;
}

double Option::payoffPath(std::vector<double> S) {
    return payoff(S[S.size() - 1]);
}

bool Option::isAsianOption() {
    return false;

}

std::vector<double> Option::getTimeSteps() {
    std::vector<double> t;
    return t;
}

bool Option::isAmericanOption() {
    return false;
}
// class Vanilla Option
VanillaOption::VanillaOption(double exp, double strike) :Option(exp) {
    if (exp < 0 || strike < 0) {
        throw attention_exception;
    }
    _strike = strike;
}

double VanillaOption::get_strike() {
    return _strike;
}

// class Call Option
CallOption::CallOption(double exp, double strike) :VanillaOption(exp, strike) {}

optionType CallOption::GetOptionType() {
    return call;
}

double CallOption::payoff(double z) {
    double h = 0;
    if (z >= get_strike()) {
        h = z - get_strike();
    }
    return h;
}

// class Put Option
PutOption::PutOption(double exp, double strike) : VanillaOption(exp, strike) {}

optionType PutOption::GetOptionType() {
    return put;
}

double PutOption::payoff(double z) {
    double h = 0;
    if (z <= get_strike()) {
        h = get_strike() - z;
    }
    return h;
}

BlackScholesPricer::BlackScholesPricer(VanillaOption* opt, double asset, double interest, double vol) {
    option = opt;
    dig_option = NULL;
    asset_price = asset;
    interest_rate = interest;
    volatility = vol;
}

BlackScholesPricer::BlackScholesPricer(DigitalOption* opt, double asset, double interest, double vol) {
    option = NULL;
    dig_option = opt;
    asset_price = asset;
    interest_rate = interest;
    volatility = vol;
}

double normalCDF(double x) // Phi(-∞, x) aka N(x)
{
    return std::erfc(-x / std::sqrt(2.0)) / 2.0;
}

double BlackScholesPricer::operator()() {


    if (option != NULL) {
        double d1 = (log(asset_price / (option->_strike)) + (interest_rate + (volatility * volatility) / 2) * option->getExpiry()) / (volatility * sqrt(option->getExpiry()));
        double d2 = d1 - (volatility * sqrt(option->getExpiry()));
        if (option->GetOptionType() == call) {
            return (asset_price * normalCDF(d1)) - ((option->_strike) * exp(-interest_rate * option->getExpiry()) * normalCDF(d2));
        }
        else {
            return option->_strike * exp(-interest_rate * option->getExpiry()) * normalCDF(-d2) - asset_price * normalCDF(-d1);
        }
    }

    else {
        double d1 = (log(asset_price / (dig_option->_strike)) + (interest_rate + (volatility * volatility) / 2) * dig_option->getExpiry()) / (volatility * sqrt(dig_option->getExpiry()));
        double d2 = d1 - (volatility * sqrt(dig_option->getExpiry()));

        if (dig_option->GetOptionType() == call) {
            return (exp(-interest_rate * dig_option->getExpiry()) * normalCDF(d2));
        }
        else {
            return (exp(-interest_rate * dig_option->getExpiry()) * normalCDF(-d2));
        }
    }

}
double normalDensity(double x) {

    return (1 / sqrt(2.0 * 2.0 * acos(0.0))) * exp(-x / 2.0);
}

double BlackScholesPricer::delta() {
    if (option != NULL) {
        double d1 = (log(asset_price / (option->_strike)) + (interest_rate + (volatility * volatility) / 2) * option->getExpiry()) / (volatility * sqrt(option->getExpiry()));
        double delta = normalCDF(d1);
        if (option->GetOptionType() == put) {
            delta = normalCDF(d1) - 1;
        }
        return delta;
    }

    else {
        double d1 = (log(asset_price / (dig_option->_strike)) + (interest_rate + (volatility * volatility) / 2) * dig_option->getExpiry()) / (volatility * sqrt(dig_option->getExpiry()));
        double d2 = d1 - (volatility * sqrt(dig_option->getExpiry()));
        if (dig_option->GetOptionType() == call) {
            return (normalDensity(d2) * exp(-interest_rate * dig_option->getExpiry())) / (volatility * asset_price * sqrt(dig_option->getExpiry()));
        }
        else {
            return (-1) * (normalDensity(d2) * exp(-interest_rate * dig_option->getExpiry())) / (volatility * asset_price * sqrt(dig_option->getExpiry()));
        }

    }


}

// class BinaryTree
template <typename T>
BinaryTree<T>::BinaryTree(std::vector <std::vector<T>> tree, int N) {
    _tree = tree;
    _depth = N;
}

template <typename T>
BinaryTree<T>::BinaryTree() {
    _depth = 0;
}

template <typename T>
void BinaryTree<T>::setDepth(int N) {
    _depth = N;
    _tree.resize(_depth + 1);
    for (int i = 0; i < _tree.size(); i++) {
        _tree[i].resize(i + 1);
    }
}

template <typename T>
void BinaryTree<T>::setNode(int i, int j, T t) {
    _tree[i][j] = t;
}

template <typename T>
T BinaryTree<T>::getNode(int i, int j) {
    return _tree[i][j];
}

template <typename T>
void BinaryTree<T>::display() {
    for (int i = 0; i < _tree.size(); i++) {
        for (int j = 0; j < _tree[i].size(); j++) {
            std::cout << _tree[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// class CRRPricer

CRRPricer::CRRPricer(Option* opt, int N, double S0, double U, double D, double R) {
    option = opt;
    depth = N;
    asset_price = S0;
    up = U;
    down = D;
    interest_rate = R;
    sigma = 0;
    //exception arbitrage
    if (!((down < interest_rate) && (interest_rate < up))) {
        std::cout << "EXCEPTION" << std::endl;
        throw attention_exception;
    }
    _tree.setDepth(depth);
    exercise.setDepth(depth);

    //isAsianOption
    if (option->isAsianOption() == true) {
        std::cout << "EXCEPTION" << std::endl;
        throw attention_exception;
    }
}
CRRPricer::CRRPricer(Option* opt, int N, double S0, double r, double sig) {
    option = opt;
    depth = N;
    asset_price = S0;

    sigma = sig;
    up = exp((interest_rate + (pow(sigma, 2) / 2)) * (option->getExpiry() / depth) + sigma * sqrt(option->getExpiry() / depth)) - 1;
    down = exp((interest_rate + (pow(sigma, 2) / 2)) * (option->getExpiry() / depth) - sigma * sqrt(option->getExpiry() / depth)) - 1;
    interest_rate = exp(r * (option->getExpiry() / depth)) - 1;

    //exception arbitrage
    if (!((down < interest_rate) && (interest_rate < up))) {
        std::cout << "EXCEPTION" << std::endl;
        throw attention_exception;
    }
    _tree.setDepth(depth);
    exercise.setDepth(depth);

    //isAsianOption
    if (option->isAsianOption() == true) {
        std::cout << "EXCEPTION" << std::endl;
        throw attention_exception;
    }
}


void CRRPricer::compute() {


    if (option->isAmericanOption()) {
        double q = (interest_rate - down) / (up - down);
        //H(S(n,i))
        BinaryTree <double> intrinsic;
        intrinsic.setDepth(depth);
        BinaryTree <double> continuation;
        continuation.setDepth(depth);
        //H(S(N,i))
        for (int i = 0; i < depth + 1; i++) {
            _tree.setNode(depth, i, option->payoff(asset_price * pow((1.0 + up), i) * pow((1.0 + down), depth - i)));
        }

        for (int n = depth - 1; n >= 0; n--) {
            for (int i = 0; i < n + 1; i++)
            {
                intrinsic.setNode(n, i, option->payoff(asset_price * pow((1.0 + up), i) * pow((1.0 + down), n - i)));
                continuation.setNode(n, i, (q * _tree.getNode(n + 1, i + 1) + (1 - q) * _tree.getNode(n + 1, i)) / (1 + interest_rate));
                _tree.setNode(n, i, std::max(intrinsic.getNode(n, i), continuation.getNode(n, i)));
                if (intrinsic.getNode(n, i) >= continuation.getNode(n, i)) exercise.setNode(n, i, true);
                else exercise.setNode(n, i, false);

            }


        }
    }
    else {
        double q = (interest_rate - down) / (up - down);
        //H(S(N,i))
        for (int i = 0; i < depth + 1; i++) {
            _tree.setNode(depth, i, option->payoff(asset_price * pow((1.0 + up), i) * pow((1.0 + down), depth - i)));
        }

        //continu value H(S(N-1...N-N-1))
        for (int n = depth - 1; n >= 0; n--) {
            for (int i = 0; i < n + 1; i++) {
                _tree.setNode(n, i, (q * _tree.getNode(n + 1, i + 1) + (1 - q) * _tree.getNode(n + 1, i)) / (1 + interest_rate));
            }
        }
    }



}

double CRRPricer::get(int n, int i) {
    return _tree.getNode(n, i);
}
double factorial(int n) {
    double fact = 1;
    if (n != 0) {
        for (int i = 1; i <= n; i++) {
            fact = fact * i;
        }
    }
    return fact;
}
double CRRPricer::operator()(bool closed_form = false) {
    double H = 0;
    if (closed_form) {
        double q = (interest_rate - down) / (up - down);

        for (int i = 0; i < depth + 1; i++) {
            H += (factorial(depth) / (factorial(i) * factorial(depth - i))) * pow(q, i) * pow(1 - q, depth - i) * option->payoff(asset_price * pow((1.0 + up), i) * pow((1.0 + down), depth - i));
        }
        H = (1 / pow((1.0 + interest_rate), depth)) * H;
    }
    else {
        compute();
        H = get(0, 0);
    }
    return H;
}

bool CRRPricer::getExercise(int i, int j) {
    return exercise.getNode(i, j);
}
// class DigitalOption
DigitalOption::DigitalOption(double exp, double strike) :Option(exp) {
    if (exp < 0 || strike < 0) {
        throw attention_exception;
    }
    _strike = strike;
}

double DigitalOption::get_strike() {
    return _strike;
}

// class DigitalCallOption
DigitalCallOption::DigitalCallOption(double exp, double strike) :DigitalOption(exp, strike) {}

optionType DigitalCallOption::GetOptionType() {
    return call;
}

double DigitalCallOption::payoff(double z) {
    double h = 0;
    if (z >= get_strike()) {
        h = 1;
    }
    return h;
}


// class DigitalPutOption

DigitalPutOption::DigitalPutOption(double exp, double strike) :DigitalOption(exp, strike) {}

optionType DigitalPutOption::GetOptionType() {
    return put;
}

double DigitalPutOption::payoff(double z) {
    double h = 0;
    if (z <= get_strike()) {
        h = 1;
    }
    return h;
}

// class AsianOption

AsianOption::AsianOption(std::vector<double> T) {
    t = T;
}



std::vector<double> AsianOption::getTimeSteps() {
    return t;
}

double AsianOption::payoffPath(std::vector<double> S) {
    double mean = 0.0;
    for (int i = 0; i < S.size(); i++) {
        mean += S[i];
    }
    mean /= S.size();
    return payoff(mean);

}

bool AsianOption::isAsianOption() {
    return true;
}

// class AsianCallOption

AsianCallOption::AsianCallOption(std::vector<double> T, double strike) :AsianOption(T) {
    _strike = strike;
}

optionType AsianCallOption::GetOptionType() {
    return call;
}

double AsianCallOption::payoff(double z) {
    return std::max(0.0, z - _strike);
}

// class AsianPutOption

AsianPutOption::AsianPutOption(std::vector<double> T, double strike) :AsianOption(T) {
    _strike = strike;
}

optionType AsianPutOption::GetOptionType() {
    return put;
}

double AsianPutOption::payoff(double z) {
    return std::max(0.0, _strike - z);
}

// class MT

MT* MT::instance = nullptr;

MT* MT::getInstance() {
    if (instance == nullptr) {
        instance = new MT();
    }
    return instance;
}

std::mt19937* MT::getGen() {
    return gen;
}

MT::MT() {
    std::random_device rd;
    gen = new std::mt19937(rd());
    std::cout << "constructeur appelé" << std::endl;

    //dis_unif = std::uniform_real_distribution<double>(0.0, 1.0);
    //dis_norm = std::normal_distribution<double>(0.0, 1.0);


}

MT::~MT() {
    delete gen;
}

double MT::rand_unif() {

    MT* Gen;
    Gen = MT::getInstance();
    std::uniform_real_distribution<> dis(0, 1);

    return dis(*Gen->getGen());
}

double MT::rand_norm() {
    MT* Gen2;
    Gen2 = MT::getInstance();
    std::normal_distribution<> dis(0, 1);

    return dis(*Gen2->getGen());
}

// class BlackScholesMCPricer

BlackScholesMCPricer::BlackScholesMCPricer(Option* opt, double So, double rate, double vol) {
    option = opt;
    initial_price = So;
    interest_rate = rate;
    volatility = vol;
    NbPaths = 0;
    estimate = 0;
    std = 0;
    sum = 0;
    sum_sq = 0;
}

int BlackScholesMCPricer::getNbPaths() {
    return NbPaths;
}
double BlackScholesMCPricer::getEstimate() {
    return estimate;
}

void BlackScholesMCPricer::generate(int nb_paths) {



    if (option->isAsianOption()) {
        NbPaths += nb_paths;
        std::vector<double> S;
        S.resize(option->getTimeSteps().size());
        S[0] = initial_price;



        for (int j = 0; j < nb_paths; j++) {

            for (int i = 1; i < S.size(); i++) {

                S[i] = S[i - 1] * exp((interest_rate - (pow(volatility, 2) / 2)) * (option->getTimeSteps()[i] - option->getTimeSteps()[i - 1]) + volatility * sqrt(option->getTimeSteps()[i] - option->getTimeSteps()[i - 1]) * MT::rand_norm());
            }

            sum += option->payoffPath(S);
            sum_sq += pow(option->payoffPath(S), 2);
        }
        std = sqrt((sum_sq / NbPaths) - pow((sum / NbPaths), 2));



        estimate = exp(-interest_rate * option->getTimeSteps()[option->getTimeSteps().size() - 1]) * sum / NbPaths;


    }
    else {
        double S;



        NbPaths += nb_paths;

        for (int j = 0; j < nb_paths; j++) {



            S = initial_price * exp((interest_rate - (pow(volatility, 2) / 2)) * (option->getExpiry()) + volatility * sqrt(option->getExpiry()) * MT::rand_norm());


            sum += option->payoff(S);
            sum_sq += pow(option->payoff(S), 2);

        }
        std = sqrt((sum_sq / NbPaths) - pow((sum / NbPaths), 2));



        estimate = exp(-interest_rate * option->getExpiry()) * sum / NbPaths;

    }
}
std::vector<double> BlackScholesMCPricer::confidenceInterval() {
    std::vector<double> IC95;
    IC95.resize(2);
    IC95[0] = estimate - 1.96 * std / sqrt(NbPaths);
    IC95[1] = estimate + 1.96 * std / sqrt(NbPaths);
    return IC95;

}

//pour utiliser dans generate on ajoute un paramètre nb_paths correspondant aux nb de paths à rajouter
double BlackScholesMCPricer::operator()() {
    if (estimate < 0) {
        throw attention_exception;
    }
    return estimate;

}

// class American Option

AmericanOption::AmericanOption(double exp, double k) : Option(exp) {
    strike = k;

}

bool AmericanOption::isAmericanOption() {
    return true;
}

double AmericanOption::getStrike() {
    return strike;
}

// class American call option

AmericanCallOption::AmericanCallOption(double exp, double k) :AmericanOption(exp, k) {}

optionType AmericanCallOption::GetOptionType() {
    return call;
}
double AmericanCallOption::payoff(double z) {
    return std::max(0.0, z - getStrike());
}

//class American put option

AmericanPutOption::AmericanPutOption(double exp, double k) :AmericanOption(exp, k) {}

optionType AmericanPutOption::GetOptionType() {
    return put;
}

double AmericanPutOption::payoff(double z) {
    return std::max(0.0, getStrike() - z);
}


// Dans le TD7 avec le BlackScholes MC Pricer, l'intervalle de confiance à 95% converge à environ 10^-2 ce qui rend la boucle très longue pour obtenir le prix avec IC>10^-2
//  (on l'a testé avec une précision à 10^-1 et ça marche)
