# Utils & helpers for TensorFlow

This is a collection of utils and helpers for TensorFlow

## TVariableInitializer
Class for initializing variables, example:

```cpp
   tfu::TVariableInitializer InitV;

   tf::Output W = InitV.Create<to::RandomUniform>(s.WithOpName("W"), {1, 10}, tf::DT_DOUBLE, {}, tf::DT_DOUBLE);
   tf::Output b = InitV.Create<to::ParameterizedTruncatedNormal>(s.WithOpName("b"), {1, 1}, tf::DT_DOUBLE, {}, 10., 1., 0., 20.);

   // ...

   tf::ClientSession Session(s);
   InitV(Session);
```
