# AutoObjectTracker

A template-based object tracker which uses image and signal processing techniques for tracking the object of interest in real time.
A mobile phone app has been developed to interface with an Arduino to control a gimbal set up which performs the object tracking in autonomously once the object of interest has been highlighted.

## Python prototype

The Python prototype of the object tracker involves particle filtering with incremental updates of the _template space_<sup>[1](http://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf)</sup> and time regression to improve performance of the tracker

## Final Project

The final app only implemented a particle filter and square-error based search for tracking due to the low computational capacity of the phone. 

### [Video of Final Project](https://youtu.be/XVJCpAkSh5U)

## Bibiliography

1. Incremental Learning for Robust Visual Tracking: David A. Ross, Jongwoo Lim, Ruei-Sung Lin 

