cd ../

# lenet
../build/tools/caffe train --solver=my_cnn_sample/alexnet_solver.prototxt
../build/tools/caffe train --solver=my_cnn_sample/lenet_auto_solver.prototxt
../build/tools/caffe train --solver=my_cnn_sample/cifar10_solver.prototxt
../build/tools/caffe train --solver=my_cnn_sample/googlenet_quick_solver.prototxt

