TOPDIR = ../

include ../definitions.make

OBJECTS = NeuralNetwork.o \
          Connection.o \
          Layer.o \
          FeatureTransformationLayer.o \
          MatrixContainer.o \
          ActivationLayer.o \
          Trainer.o \
          Forwarder.o \
          FeatureTransformation.o \
          MinibatchGenerator.o \
          Statistics.o \
          Estimator.o \
          Regularizer.o \
          GradientBasedTrainer.o \
          LearningRateSchedule.o \
          TrainingCriteria.o \
          MultiPortLayer.o 

OBJ = $(patsubst %, objects/%, $(OBJECTS))

LIB = ../Core/libCore.a \
      ../Math/libMath.a \
      ../Features/libFeatures.a

.PHONY: all prepare clean Application libNeuralNetwork.a

all: prepare $(OBJ) libNeuralNetwork.a Application

prepare:
	@mkdir -p objects

$(OBJ): objects/%.o : %.cc %.hh
	$(CC) $(COPTS) -c $< -lcppunit -o $@

libNeuralNetwork.a:
	$(MAKELIB) $(ARFLAGS) $@ $(OBJ)

Application: $(OBJ)
	$(CC) $(COPTS) $@.cc $(OBJ) -Wl,--start-group $(LIB) -Wl,--end-group $(CLIB) -o neural-network-trainer

clean:
	rm -rf objects/ libNeuralNetwork.a neural-network-trainer
