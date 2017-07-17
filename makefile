SHELL=/bin/bash

CC = g++
SRCDIR := src
BUILDDIR := build
TARGET := bin

CPPFLAGS = -std=c++11
CPPFLAGS += -W -Wall
CPPFLAGS += -O3
CPPFLAGS += -I include
CPPFLAGS += -Wno-unused-parameter
CPPFLAGS += -Wno-deprecated-declarations
CPPFLAGS += -Wno-deprecated-declarations
CPPFLAGS += -Wno-ignored-qualifiers
CPPFLAGS += -Wno-unused-command-line-argument
CPPFLAGS += -Wno-inconsistent-missing-override



UNAME_S := $(shell uname -s)
	 ifeq ($(UNAME_S),Linux)
			 CPPFLAGS += -lOpenCL
	 endif
	 ifeq ($(UNAME_S),Darwin)
			 CPPFLAGS += -framework OpenCL
	 endif

LDLIBS = -ltbb
LDLIBS +=-lboost_timer
LDLIBS +=-lboost_system

SRCEXT := cpp

SUBDIRS := $(wildcard */)
SOURCES := src/optionPricer.cpp src/pricerRegistrar.cpp
SOURCES+= $(wildcard $(SRCDIR)/**/*.cpp)
SOURCES:=$(SOURCES:$(SRCDIR)/%=%)

#$(info SOURCES found: $(SOURCES))

OBJECTS=$(patsubst %, build/%, ${SOURCES:.cpp=.o})

EXECUTABLES=simplisticTest specificSimulation

all: $(EXECUTABLES)


$(EXECUTABLES): $(OBJECTS)
	@mkdir -p $(TARGET)
	$(CC) $(CPPFLAGS) $(LDLIBS) $(OBJECTS) -o $(TARGET)/$@ $(SRCDIR)/$@.$(SRCEXT)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	 @mkdir -p $(BUILDDIR)
	 @mkdir -p $(BUILDDIR)/engines
	 $(CC) $(CPPFLAGS) -c -o $@ $<

clean:
	@echo "Cleaning...";
	rm -f $(OBJECTS) $(TARGET)/*
