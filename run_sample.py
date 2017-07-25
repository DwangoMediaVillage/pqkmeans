import pqkmeans

e = pqkmeans.encoder.EncoderSample()
e.fit_generator(([i] for i in range(10)))

for value in e.transform_generator(([i] for i in range(10) for _ in range(3))):
    print(value)