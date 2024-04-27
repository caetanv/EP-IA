
def create_mlp_design():
    # Widgets for MLP design
    input_size_widget = widgets.IntSlider(value=120, min=1, max=1000, step=1, description='Input Size:')
    hidden_size_widget = widgets.IntSlider(value=50, min=1, max=500, step=1, description='Hidden Size:')
    learning_rate_widget = widgets.FloatSlider(value=0.1, min=0.01, max=1.0, step=0.01, description='Learning Rate:')
    epochs_widget = widgets.IntSlider(value=1000, min=100, max=5000, step=100, description='Epochs:')
    early_stopping_widget = widgets.Checkbox(value=False, description='Early Stopping')
    cross_validation_widget = widgets.Checkbox(value=False, description='Cross-Validation')
    num_folds_widget = widgets.IntSlider(value=5, min=2, max=10, step=1, description='Num Folds:')
    train_button = widgets.Button(description='Train MLP')
    load_weights_button = widgets.FileUpload(description='Load Weights', accept='.csv')
    save_weights_button = widgets.Button(description='Save Weights')

    # Display widgets
    display(input_size_widget, hidden_size_widget, learning_rate_widget, epochs_widget,
            early_stopping_widget, cross_validation_widget, num_folds_widget,
            train_button, load_weights_button, save_weights_button)

    mlp = None

    def train_mlp(button):
        nonlocal mlp
        input_size = input_size_widget.value
        hidden_size = hidden_size_widget.value
        learning_rate = learning_rate_widget.value
        epochs = epochs_widget.value
        early_stopping = early_stopping_widget.value
        use_cross_validation = cross_validation_widget.value
        num_folds = num_folds_widget.value

        mlp = MLP(input_size, hidden_size, output_size=1)  # Adjust output_size as needed
        mlp.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate,
                  early_stopping=early_stopping, use_cross_validation=use_cross_validation, num_folds=num_folds)

    def load_weights(change):
        for filename in load_weights_button.value:
            content = load_weights_button.value[filename]['content']
            with open(filename, 'wb') as f:
                f.write(content)
            mlp.load_weights(filename)

    def save_weights(button):
        if mlp is not None:
            filename = 'pesos.csv'
            mlp.save_weights(filename)

    # Attach event handlers to buttons
    #train_button.on

create_mlp_design()