
COMMON_FLAGS=-Wall -Wextra -Wpedantic -std=c++17
LINKER_FLAGS=-lpthread -lOpenCL
ifdef DEBUG
	CPPFLAGS=${COMMON_FLAGS} -g -O0
else
	CPPFLAGS=${COMMON_FLAGS} -O3 -DNDEBUG
endif

SOURCES= \
	src/main.cpp \
	src/blue_noise.cpp
OBJECTS=${subst .cpp,.o,${SOURCES}}

all: Dithering

Dithering: ${OBJECTS}
	${CXX} ${CPPFLAGS} ${LINKER_FLAGS} -o Dithering $^

.PHONY:

clean:
	rm -f Dithering
	rm -f ${OBJECTS}
