import pandas as pd
import tensorflow as tf


def main():
    a = tf.constant(1)
    b = tf.constant(1)
    c = tf.add(a, b)
    print(c)


if __name__ == "__main__":
    main()
