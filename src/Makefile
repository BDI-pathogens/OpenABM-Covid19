CC=gcc
CFLAGS=-I.
OBJ = main.o

# Name of executable
_EXE = covid19ibm.exe
EXE = $(_EXE)


%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)


all: ${OBJ}
	${CC} -o ${EXE} ${OBJ}

clean:
	rm -f $(OBJ) $(EXE)
