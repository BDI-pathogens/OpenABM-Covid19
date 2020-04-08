#include "swig_utils.h"


/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/
int util_n_current( model *model, int type ) {
    return model->event_lists[type].n_current;
}

int util_n_total( model *model, int type ) {
    return model->event_lists[type].n_total;
}

int util_n_total_age( model *model, int type, int age ) {
    return model->event_lists[type].n_total_by_age[age];
}

int util_n_daily( model *model, int type, int day ) {
    return model->event_lists[type].n_daily_current[day];
}
