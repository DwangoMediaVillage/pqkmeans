import pqkmeans


def data_source():
    return ([i*100] for i in range(10) for _ in range(3))


e = pqkmeans.encoder.EncoderSample()
e.fit_generator(data_source())

for original, encoded, decoded in zip(
        data_source(),
        e.transform_generator(data_source()),
        e.inverse_transform_generator(e.transform_generator(data_source()))
):
    print("{} -> {} -> {}".format(original, encoded, decoded))
