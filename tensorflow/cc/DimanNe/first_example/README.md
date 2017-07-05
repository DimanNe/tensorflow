# First example

This is an example of the very basics of TensorFlow:
1. Variable initializing with `tfu::TVariableInitializer`:
```cpp
   tfu::TVariableInitializer InitV;
   tf::Output W = InitV.Create<to::RandomUniform>(s.WithOpName("W"), {1, 10}, tf::DT_DOUBLE, {}, tf::DT_DOUBLE);
   tf::Output b = InitV.Create<to::ParameterizedTruncatedNormal>(s.WithOpName("b"), {1, 1}, tf::DT_DOUBLE, {}, 10., 1., 0., 20.);
   // ...
   tf::ClientSession Session(s);
   InitV(Session);

```
2. Constructing computational graph consisting of basic operations (matrix/vector multiplication)
