%module model_utils

%inline %{
int utils_n_current( model *pmodel, int type ) {
    return pmodel->event_lists[type].n_current;
}

int utils_n_total( model *pmodel, int type ) {
    return pmodel->event_lists[type].n_total;
}

int utils_n_total_by_day( model *pmodel, int type, int day ) {
    return pmodel->event_lists[type].n_daily[day];
}

int utils_n_total_age( model *pmodel, int type, int age ) {
    return pmodel->event_lists[type].n_total_by_age[age];
}

int utils_n_daily( model *pmodel, int type, int day ) {
    return pmodel->event_lists[type].n_daily_current[day];
}

int utils_n_daily_age( model *pmodel, int type, int day, int age) {
    return pmodel->event_lists[type].n_daily_by_age[day][age];
}


%}


%extend model{
    ~model() {
        destroy_model($self);
    }
}

/* These destructors shouldn't be used directly (use create_model and
 * destroy_model). However, SWIG automatically generates R ctors and dtors
 * for all structs that it can find. This causes `R CMD check` to output
 * NOTEs such as this:
 *
 *   directory: no visible binding for global variable 'delete_directory'
 *
 * So add dtors to suppress these messages. The generated SWIG ctor internally
 * uses calloc(3), so use the free(3) function.
 */
%extend directory{
    ~directory() {
        free($self);
    }
}
%extend event{
    ~event() {
        free($self);
    }
}
%extend event_list{
    ~event_list() {
        free($self);
    }
}
%extend individual{
    ~individual() {
        free($self);
    }
}
%extend infection_event{
    ~infection_event() {
        free($self);
    }
}
%extend interaction{
    ~interaction() {
        free($self);
    }
}
%extend interaction_block{
    ~interaction_block() {
        free($self);
    }
}
