#ifndef LM_APPLICATION_HH_
#define LM_APPLICATION_HH_

#include "Core/CommonHeaders.hh"
#include "Core/Application.hh"

namespace Lm {

class Application: public Core::Application
{
private:
	static const Core::ParameterEnum paramAction_;
	static const Core::ParameterString paramCorpus_;
	enum Actions { none, build, perplexity, generateSentences };

	void trainLanguageModel();
	void computePerplexity();
public:
	virtual ~Application() {}
	virtual void main();
};

} // namespace

#endif /* LM_APPLICATION_HH_ */
