#include <stdio.h>
#include <string.h>

#define OK       0
#define NO_INPUT 1
#define TOO_LONG 2

static int getLine (char *prmpt, char *buff, size_t sz) {
  int ch, extra;

  // Get line with buffer overrun protection.
  if (prmpt != NULL) {
    printf ("%s", prmpt);
    fflush (stdout);
  }
  if (fgets (buff, sz, stdin) == NULL)
    return NO_INPUT;

  // If it was too long, there'll be no newline. In that case, we flush
  // to end of line so that excess doesn't affect the next call.
  if (buff[strlen(buff)-1] != '\n') {
    extra = 0;
    while (((ch = getchar()) != '\n') && (ch != EOF))
    extra = 1;
    return (extra == 1) ? TOO_LONG : OK;
  }

  // Otherwise remove newline and give string back to caller.
  buff[strlen(buff)-1] = '\0';
  return OK;
}

int main (void) {
  int rc;
  char buff[10];

  rc = getLine ("Enter string> ", buff, sizeof(buff));
  if (rc == NO_INPUT) {
  
    // Extra NL since my system doesn't output that on EOF.
    printf ("No input\n");
    return 1;
  }

  if (rc == TOO_LONG) {
    printf ("Input too long [%s]\n", buff);
    return 1;
  }

  printf ("OK [%s]\n", buff);

  return 0;
}
