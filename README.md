# MorF
Gender classifier built with the Tensorflow and Keras API, and trained with 20,000 names + genders from [BehindTheName](https://www.behindthename.com/).
This is project for my own learning, meaning as I learn more about machine learning, I will (ideally) update the classifier and improve its' accuracy.

<!-- Status indicators -->
## Status
Last recorded model accuracy: 77.5% ~~74.5~~ ~~70.5~~ (55% better than randomly guessing)
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

<!-- USAGE EXAMPLES -->
## Usage

```sh
python3 run.py data.csv
```
`data.csv` is a CSV file with headers `name` and `gender`. An `F` corresponds to female, and an
`M` corresponds to male. These must be ***first names*** only, and contain at least one name-gender entry.


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
