# vim: ft=pyrex

# The following is mostly stolen from ecyglpki, thanks!

cdef extern from "glpk.h":

    enum:
        MAJOR_VERSION "GLP_MAJOR_VERSION"
        MINOR_VERSION "GLP_MINOR_VERSION"

    enum:
        ON  "GLP_ON"   # copy names
        OFF "GLP_OFF"  # don't copy names

    enum:
        MIN "GLP_MIN"  #  minimization
        MAX "GLP_MAX"  #  maximization

    enum:
        FR "GLP_FR"  #  free (unbounded) variable
        LO "GLP_LO"  #  variable with lower bound
        UP "GLP_UP"  #  variable with upper bound
        DB "GLP_DB"  #  double-bounded variable
        FX "GLP_FX"  #  fixed variable

    enum:
        UNDEF  "GLP_UNDEF"  #  solution is undefined
        FEAS   "GLP_FEAS"   #  solution is feasible
        INFEAS "GLP_INFEAS" #  solution is infeasible
        NOFEAS "GLP_NOFEAS" #  no feasible solution exists
        OPT    "GLP_OPT"    #  solution is optimal
        UNBND  "GLP_UNBND"  #  solution is unbounded

    enum:
        MSG_OFF "GLP_MSG_OFF"  #  no output
        MSG_ERR "GLP_MSG_ERR"  #  warning and error messages only
        MSG_ON  "GLP_MSG_ON"   #  normal output
        MSG_ALL "GLP_MSG_ALL"  #  full output
        MSG_DBG "GLP_MSG_DBG"  #  debug output

    enum:
        PRIMAL "GLP_PRIMAL" #  use primal simplex
        DUALP  "GLP_DUALP"  #  use dual if it fails, use primal
        DUAL   "GLP_DUAL"   #  use dual simplex

    cdef struct Prob "glp_prob":
        pass

    ctypedef struct SmCp "glp_smcp":
        int msg_lev     #  message level
        int meth        #  simplex method option

    # manage problem

    Prob* create_prob "glp_create_prob" ()
    void delete_prob "glp_delete_prob" (Prob* prob)
    void copy_prob "glp_copy_prob" (Prob* dest, Prob* prob, int names)

    # manage constraint matrix

    int add_rows "glp_add_rows" (Prob* prob, int rows)
    int add_cols "glp_add_cols" (Prob* prob, int cols)

    void del_rows "glp_del_rows" (Prob* prob, int rows, const int row_ind[])
    void del_cols "glp_del_cols" (Prob* prob, int cols, const int col_ind[])

    void set_row_bnds "glp_set_row_bnds" (Prob* prob, int row, int vartype,
                                          double lb, double ub)
    void set_col_bnds "glp_set_col_bnds" (Prob* prob, int col, int vartype,
                                          double lb, double ub)

    void set_mat_row "glp_set_mat_row" (Prob* prob, int row, int length,
                                        const int ind[], const double val[])
    void set_mat_col "glp_set_mat_col" (Prob* prob, int col, int length,
                                        const int ind[], const double val[])

    # access constraint matrix

    int get_num_rows "glp_get_num_rows" (Prob* prob)
    int get_num_cols "glp_get_num_cols" (Prob* prob)

    double get_row_lb "glp_get_row_lb" (Prob* prob, int row)
    double get_row_ub "glp_get_row_ub" (Prob* prob, int row)

    double get_col_lb "glp_get_col_lb" (Prob* prob, int col)
    double get_col_ub "glp_get_col_ub" (Prob* prob, int col)

    int get_mat_row "glp_get_mat_row" (Prob* prob, int row,
                                       int ind[], double val[])
    int get_mat_col "glp_get_mat_col" (Prob* prob, int col,
                                       int ind[], double val[])

    # objective function

    void set_obj_dir "glp_set_obj_dir" (Prob* prob, int optdir)
    void set_obj_coef "glp_set_obj_coef" (Prob* prob, int col, double coef)

    int get_obj_dir "glp_get_obj_dir" (Prob* prob)
    double get_obj_coef "glp_get_obj_coef" (Prob* prob, int col)
    double get_obj_val "glp_get_obj_val" (Prob* prob)

    # solve

    void std_basis "glp_std_basis" (Prob* prob)
    void init_smcp "glp_init_smcp" (SmCp* cp)
    int simplex "glp_simplex" (Prob* prob, const SmCp* cp)

    # access solution

    int get_status "glp_get_status" (Prob* prob)
    int get_prim_stat "glp_get_prim_stat" (Prob* prob)
    int get_dual_stat "glp_get_dual_stat" (Prob* prob)

    double get_row_prim "glp_get_row_prim" (Prob* prob, int row)
    double get_row_dual "glp_get_row_dual" (Prob* prob, int row)

    double get_col_prim "glp_get_col_prim" (Prob* prob, int col)
    double get_col_dual "glp_get_col_dual" (Prob* prob, int col)

    # names

    void set_prob_name "glp_set_prob_name" (Prob* P, const char* name)
    void set_obj_name "glp_set_obj_name" (Prob* P, const char* name)

    const char* get_prob_name "glp_get_prob_name" (Prob* P)
    const char* get_obj_name "glp_get_obj_name" (Prob* P)

    void set_row_name "glp_set_row_name" (Prob* P, int i, const char *name)
    void set_col_name "glp_set_col_name" (Prob* P, int i, const char *name)

    const char* get_row_name "glp_get_row_name" (Prob* P, int i)
    const char* get_col_name "glp_get_col_name" (Prob* P, int i)
