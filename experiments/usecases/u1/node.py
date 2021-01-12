import mmlib


def main():
    # wait for new model to be ready
    model_info = listen_for_models()
    # as soon as new model is available
    recovered_model = mmlib.recover.recover_model(model_info)
    # use recovered model
    mmlib.log.use_model(recovered_model)


if __name__ == '__main__':
    main()