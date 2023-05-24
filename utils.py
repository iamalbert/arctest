from typing import NamedTuple

def size_of(it):
    return len(tuple(it))


class Result(NamedTuple):
    acc: float
    f1: float
    precision:float
    recall: float

    def __str__(self):
        return f"Result(acc={self.acc:.4f}, f1={self.f1:.4f}, precision={self.precision:.4f}, recall={self.recall:.4f})"

def acc_and_f1(y_true, y_pred) -> Result:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return Result(
        accuracy_score(y_true, y_pred), 
        f1_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        )


def escape_lightgbm(string):

    escape_char = r'&\'-:()/ '

    trans = {
        ord(c): ord('_') for c in escape_char
    }

    return string.translate(trans)

