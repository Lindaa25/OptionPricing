#pragma once
#pragma once
#include <vector>
#include <exception>
#include <random>


class condition_invalid_exception :public std::exception {
	const char* what() const throw() {
		return "Attention exception";
	}
}attention_exception;


class Option {
public:
	Option();
	Option(double _expiry);
	virtual double payoff(double) = 0;
	double getExpiry();

	virtual double payoffPath(std::vector<double>);
	virtual bool isAsianOption();
	virtual bool isAmericanOption();
	virtual std::vector<double> getTimeSteps();


private:
	double _expiry;
};

enum optionType { call, put };

//classe dérivée de Option
class VanillaOption :public Option {
public:
	VanillaOption(double _expiry, double _strike);
	virtual optionType GetOptionType() = 0;
	double get_strike();
private:
	double _strike;
	friend class BlackScholesPricer;
};

class CallOption :public VanillaOption {
public:
	CallOption(double _expiry, double _strike);
	optionType GetOptionType() override;
	double payoff(double) override;
};


class PutOption :public VanillaOption {
public:
	PutOption(double _expiry, double _strike);
	optionType GetOptionType() override;
	double payoff(double) override;

};

// classe dérivée de Option
class DigitalOption : public Option {
public:
	DigitalOption(double _expiry, double _strike);
	virtual optionType GetOptionType() = 0;
	double get_strike();

private:
	double _strike;
	friend class BlackScholesPricer;
};

// classe dérivée de DigitalOption
class DigitalCallOption :public DigitalOption {
public:
	DigitalCallOption(double _expiry, double _strike);
	optionType GetOptionType() override;
	double payoff(double) override;
};

// classe dérivée de DigitalOption
class DigitalPutOption :public DigitalOption {
public:
	DigitalPutOption(double _expiry, double _strike);
	optionType GetOptionType() override;
	double payoff(double) override;

};

class BlackScholesPricer {
public:
	BlackScholesPricer(VanillaOption* option, double asset_price, double interest_rate, double volatility);
	BlackScholesPricer(DigitalOption* dig_option, double asset_price, double interest_rate, double volatility);

	double operator()();
	double delta();

private:
	VanillaOption* option;
	DigitalOption* dig_option;
	double asset_price;
	double interest_rate;
	double volatility;
};

//pour avoir T prend n'importe quel type
template <typename T>

class BinaryTree {
public:
	BinaryTree();
	BinaryTree(std::vector <std::vector<T>> _tree, int _depth);
	void setDepth(int);
	void setNode(int, int, T);
	T getNode(int, int);
	void display();
private:
	std::vector < std::vector<T> > _tree;
	int _depth;
};

class CRRPricer {
public:
	CRRPricer(Option* option, int depth, double asset_price, double up, double down, double interest_rate);
	CRRPricer(Option* option, int depth, double asset_price, double interest_rate, double sigma);

	void compute();
	double get(int, int);
	double operator()(bool closed_form);
	bool getExercise(int, int);
private:
	Option* option;
	int depth;
	double asset_price;
	double up;
	double down;
	double interest_rate;
	double sigma;
	BinaryTree <double> _tree;
	BinaryTree<bool> exercise;
};

class AsianOption :public Option {
public:
	AsianOption(std::vector<double> t);
	std::vector<double> getTimeSteps() override;
	virtual optionType GetOptionType() = 0;
	double payoffPath(std::vector<double>) override;

	bool isAsianOption() override;

private:
	std::vector<double> t;

};

class AsianCallOption : public AsianOption {
public:
	AsianCallOption(std::vector<double> t, double _strike);
	optionType GetOptionType() override;
	double payoff(double) override;

private:
	double _strike;
};

class AsianPutOption : public AsianOption {
public:
	AsianPutOption(std::vector<double> t, double _strike);
	optionType GetOptionType() override;
	double payoff(double) override;

private:
	double _strike;
};

class MT {
public:
	~MT();
	void operator=(const MT&) = delete;
	static MT* getInstance();
	static double rand_unif();
	static double rand_norm();
	std::mt19937* getGen();


private:
	MT();
	static MT* instance;

	std::mt19937* gen;
	//std::uniform_real_distribution<> dis_unif;
	//std::normal_distribution<> dis_norm;

};


class BlackScholesMCPricer {
public:
	BlackScholesMCPricer(Option* option, double initial_price, double interest_rate, double volatility);
	int getNbPaths();
	void generate(int NbPaths);
	double operator()();
	double getEstimate();
	std::vector<double> confidenceInterval();

private:
	Option* option;
	double initial_price;
	double interest_rate;
	double volatility;
	int NbPaths;
	double estimate;
	double std;
	double sum;
	double sum_sq;

};

class AmericanOption : public Option {

public:
	AmericanOption(double expiry, double strike);
	bool isAmericanOption() override;
	virtual optionType GetOptionType() = 0;
	double getStrike();
private:
	double strike;

};

class AmericanCallOption : public AmericanOption {
public:

	AmericanCallOption(double expiry, double strike);
	optionType GetOptionType() override;
	double payoff(double) override;



};

class AmericanPutOption : public AmericanOption {

public:
	AmericanPutOption(double expiry, double strike);
	optionType GetOptionType() override;
	double payoff(double) override;

private:

};