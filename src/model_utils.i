%module model_utils

#include "model.h"

%inline %{
int utils_n_current( model *model, int type ) {
    return model->event_lists[type].n_current;
}

int utils_n_total( model *model, int type ) {
    return model->event_lists[type].n_total;
}

int utils_n_total_age( model *model, int type, int age ) {
    return model->event_lists[type].n_total_by_age[age];
}

int utils_n_daily( model *model, int type, int day ) {
    return model->event_lists[type].n_daily_current[day];
}
%}


%extend model{
    ~model() {
        destory_model($self);
    }
}
