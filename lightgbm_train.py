import lightgbm as lgb
from sklearn import metrics
import random
from sklearn.datasets import load_svmlight_file


def train():
    train_set = lgb.Dataset('model/train.txt')
    params = {'objective': 'xentropy', 'learning_rate': 0.1, 'num_threads': 4}
    model = lgb.train(params=params, train_set=train_set, num_boost_round=1)
    model.save_model(filename='model/lgb.model')


def test():
    model = lgb.Booster(model_file='model/lgb.model')
    y_pred = model.predict('model/valid.txt')
    y_pred = map(lambda x: 1 if x > 0.5 else 0, y_pred)
    y_true = []
    with open('model/valid.txt', 'r') as valid_file:
        for line in valid_file:
            y_true.append(int(line[0]))

    y_pred = [random.randint(0, 1) for i in range(len(y_pred))]
    print metrics.accuracy_score(y_true=y_true, y_pred=y_pred)


def main():
    # train()
    test()


if __name__ == '__main__':
    main()
