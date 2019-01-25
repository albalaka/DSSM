def run():
    """
    Create and train a model
    """

    with tf.Session as sess:
        model = DeepStateSpaceModel(sess)

        model.build_model().build_loss().initialize_variables()

        error = model.train()

        return error

if __name__ == '__main__':
    error = run()
