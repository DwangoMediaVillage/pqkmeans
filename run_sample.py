import pqkmeans


def data_source(n: int):
    for i in range(n):
        for _ in range(3):
            yield [i * 100]


e = pqkmeans.encoder.EncoderSample()
e.fit_generator(data_source(20))

# can handle infinite list
inf = 1000000000
for i, original, encoded, decoded in zip(
        range(inf),
        data_source(inf),
        e.transform_generator(data_source(inf)),
        e.inverse_transform_generator(e.transform_generator(data_source(inf)))
):
    print("{} -> {} -> {}".format(original, encoded, decoded))
    if i == 10:
        break
