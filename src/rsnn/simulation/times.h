#include <stdlib.h>

typedef struct time {
    double s;
    struct time *prev;
};

void init_time(struct time **ptr);
void append_time(struct time **ptr, double s);
void pop_time(struct time **ptr);
void clean_time(struct time **ptr);

