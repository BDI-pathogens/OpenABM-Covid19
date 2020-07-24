%module model_utils

%inline %{
int utils_n_current( model *pmodel, int type ) {
    return pmodel->event_lists[type].n_current;
}

int utils_n_total( model *pmodel, int type ) {
    return pmodel->event_lists[type].n_total;
}

int utils_n_total_age( model *pmodel, int type, int age ) {
    return pmodel->event_lists[type].n_total_by_age[age];
}

int utils_n_daily( model *pmodel, int type, int day ) {
    return pmodel->event_lists[type].n_daily_current[day];
}
%}


%extend model{
    ~model() {
        destroy_model($self);
    }
}
