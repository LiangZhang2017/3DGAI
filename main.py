
import argparse
from config import model_config
'''
Start one example dataset
'''
class KCModel:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Arguments Parameters Inputs')
        parser.add_argument("--data_path", nargs=1, type=str, default=['/dataset'])
        parser.add_argument("--Course", nargs=1, type=str, default=['CSAL'])
        parser.add_argument("--Lesson_Id", nargs=2, type=str, default=['Lesson_Id', 'lesson1'])
        parser.add_argument('--LearningModel', nargs=1, type=str, default=['MTF'])
        args = parser.parse_args()
        self.args = args

    def main(self):
        config=model_config(self.args)
        config.main()

if __name__ == '__main__':
    obj = KCModel()
    obj.main()