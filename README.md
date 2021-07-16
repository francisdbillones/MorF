# MorF
Gender classifier built with the Tensorflow and Keras API, and trained with 95,000 names + genders from [Data World](https://data.world/howarder/gender-by-name).
This is project for my own learning, meaning as I learn more about machine learning, I will (ideally) update the classifier and improve its' accuracy.

<!-- Status indicators -->
## Status
Last recorded model accuracy: 90.2% ~~79%~~ ~~77.5%~~ ~~74.5~~ ~~70.5~~ (80.4% better than randomly guessing)
<br>


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/francisdbillones/MorF.git
   ```
2. Install Python packages
   ```sh
   python3 -m pip install -r requirements.txt
   ```

### Training with your own data
If you want to train the model with your own data, pass an argument to `MorF.py`:
```sh
python3 MorF.py data.csv
```

`data.csv` is a CSV file with headers `name` and `gender`. A 1 corresponds to female, and a
0 corresponds to male. These must be ***first names*** only, and contain at least one name-gender entry.

<!-- USAGE EXAMPLES -->
## Using the pre-trained model

The model is provided through the `model` directory. <br>
Load it into memory using Keras:
```py
import tensorflow.keras as keras
model = keras.models.load_model('model')
```


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
