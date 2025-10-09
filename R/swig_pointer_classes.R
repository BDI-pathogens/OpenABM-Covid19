# R/swig_pointer_classes.R
# Define SWIG pointer proxy S4 classes that the wrapper expects.
# Add any other _p_* classes you see referenced by the generated wrapper.

# char *
if (!isClass("_p_char")) {
  setClass("_p_char", representation(ref = "character"))
}

# network *
if (!isClass("_p_network")) {
  setClass("_p_network", representation(ref = "externalptr"))
}
