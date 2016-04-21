#ifndef CORE_APPLICATION_HH_
#define CORE_APPLICATION_HH_

#define APPLICATION(A)                          \
	int main(int argc, const char* argv[]) {    \
		A app;                                  \
		app.run(argc, argv);                    \
		return 0;                               \
	}

// handler for segmentation faults
void handler(int sig);

namespace Core {

class Application
{
private:
public:
	virtual ~Application() {}
	virtual void run(int argc, const char* argv[]);
	virtual void main() {};
};

} // namespace

#endif /* CORE_APPLICATION_HH_ */
