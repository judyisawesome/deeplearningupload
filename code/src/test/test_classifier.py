def test_data_shape(iris_data):
    X,y=iris_data
    assert X.shape[1]=4
    assert len(np.unique(y))==3
def test_model_accuracy(model,split_data):
    X_train, X_test, y_train, y_test= split_data
    y_train_ohe=to_categorical(y_train)
