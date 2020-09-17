%module model_utils

%inline %{
int utils_n_current( model *model, int type ) {
    return model->event_lists[type].n_current;
}

int utils_n_total( model *model, int type ) {
    return model->event_lists[type].n_total;
}

int utils_n_total_by_day( model *model, int type, int day ) {
    return model->event_lists[type].n_daily[day];
}

int utils_n_total_age( model *model, int type, int age ) {
    return model->event_lists[type].n_total_by_age[age];
}

int utils_n_daily( model *model, int type, int day ) {
    return model->event_lists[type].n_daily_current[day];
}

int utils_n_daily_age( model *model, int type, int day, int age) {
    return model->event_lists[type].n_daily_by_age[day][age];
}


%}


%extend model{
    ~model() {
        destroy_model($self);
    }
}
