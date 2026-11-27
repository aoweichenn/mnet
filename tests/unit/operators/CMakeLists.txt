mnet_add_operator_tests(activation activation/pointwise_tests.cpp activation/situ_tests.cpp)
mnet_add_operator_tests(arithmetic arithmetic/broadcast_tests.cpp arithmetic/reduction_tests.cpp
    arithmetic/aggregation_tests.cpp)
mnet_add_operator_tests(convolution convolution/border_tests.cpp convolution/convolution_tests.cpp
    convolution/variants_tests.cpp convolution/spatial_tests.cpp convolution/transposed_tests.cpp
    convolution/scalar_tests.cpp)
mnet_add_operator_tests(indexing indexing/gather_scatter_tests.cpp indexing/boundary_tests.cpp)
mnet_add_operator_tests(linear_algebra linear_algebra/matrix_tests.cpp linear_algebra/weighted_tests.cpp)
mnet_add_operator_tests(normalization normalization/normalization_tests.cpp normalization/affine_tests.cpp
    normalization/statistics_tests.cpp)
mnet_add_operator_tests(sequence sequence/attention_tests.cpp sequence/attention_mask_tests.cpp
    sequence/state_tests.cpp sequence/recurrent_tests.cpp)
mnet_add_operator_tests(shape shape/dynamic_tests.cpp shape/layout_tests.cpp
    shape/manipulation_tests.cpp shape/onnx_layout_tests.cpp shape/rearrangement_tests.cpp)
mnet_add_operator_tests(signal signal/stft_tests.cpp signal/spectrogram_tests.cpp)
mnet_add_operator_tests(vision vision/pooling_tests.cpp vision/pooling_variants_tests.cpp vision/sampling_tests.cpp)
mnet_add_operator_tests(detection detection/ssd_tests.cpp detection/yolo_tests.cpp)
mnet_add_operator_tests(quantization quantization/conversion_tests.cpp)
mnet_add_operator_tests(sources sources/source_tests.cpp)
