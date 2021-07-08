# MorF
Gender classifier built with the Tensorflow and Keras API, and trained with 20,000 names + genders from [BehindTheName](https://www.behindthename.com/).
This is project for my own learning, meaning as I learn more about machine learning, I will (ideally) update the classifier and improve its' accuracy.

<!-- Status indicators -->
## Status
Last recorded model accuracy: 74.5% ~~70.5%~~ (49% better than randomly guessing)
<br>
Code status: refactoring


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
python3 run.py data_dir/
```
`data_dir/` is a directory that must contain `male_names.txt`, `female_names.txt`, and `androgynous_names.txt`. These files must contain line separated ***first names*** only. It is okay for any of these files to be empty.


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.md` for more information.
