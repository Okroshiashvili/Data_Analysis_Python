import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
)
from sklearn.exceptions import NotFittedError


def confusion_plot(matrix, labels=None):
    """
    Display confusion matrix as heatmap

    """

    labels = labels if labels else ["Negative (0)", "Positive (1)"]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(
        data=matrix,
        cmap="Blues",
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_xlabel("PREDICTED")
    ax.set_ylabel("ACTUAL")
    ax.set_title("Confusion Matrix")
    plt.close()

    return fig


def plot_roc(y_true, y_probs, label, compare=False, ax=None):
    """
    Plot ROC Curve.

    Set compare=True to compare classifiers.
    """

    fpr, tpr, thresh = roc_curve(y_true, y_probs)
    auc = round(roc_auc_score(y_true, y_probs), 2)

    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)
    label = " ".join([label, f"({auc})"]) if compare else None

    sns.lineplot(x=fpr, y=tpr, ax=axis, label=label)

    if compare:
        axis.legend(title="Classifier (AUC)", loc="lower right")
    else:
        axis.text(
            0.72,
            0.05,
            f"AUC = {auc}",
            fontsize=12,
            bbox=dict(facecolor="green", alpha=0.4, pad=5),
        )

        axis.fill_between(
            fpr, fpr, tpr, alpha=0.3, edgecolor="g", linestyle="--", linewidth=2
        )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    plt.close()

    return axis if ax else fig


def plot_precision_recall(y_true, y_probs, label, compare=False, ax=None):
    """
    Plot Precision-Recall Curve

    Set compare=True to compare classifiers.

    """

    p, r, thresh = precision_recall_curve(y_true, y_probs)
    p, r, thresh = list(p), list(r), list(thresh)

    p.pop()
    r.pop()

    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)

    if compare:
        sns.lineplot(x=r, y=p, ax=axis, label=label)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.legend(loc="lower left")

    else:
        sns.lineplot(x=thresh, y=p, label="Precision", ax=axis)
        axis.set_xlabel("Threshold")
        axis.set_ylabel("Precision")
        axis.legend(loc="lower left")

        axis_twin = axis.twinx()

        sns.lineplot(x=thresh, y=r, color="limegreen", label="Recall", ax=axis_twin)

        axis_twin.set_ylabel("Recall")
        axis_twin.set_ylim(0, 1)
        axis_twin.legend(bbox_to_anchor=(0.24, 0.18))

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("Precision Vs Recall")

    plt.close()

    return axis if ax else fig


def plot_feature_importance(importance, feature_labels, ax=None):
    """
    Plot feature importance

    """

    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1, figsize=(5, 10))

    sns.barplot(x=importance, y=feature_labels, ax=axis)

    axis.set_title("Feature Importance")

    plt.close()

    return axis if ax else fig


def train_classifier(clf, x_train, y_train, sample_weight=None, refit=False):
    """
    Trains given classifier.

    Moreover, calculates training accuracy.
    """

    try:
        if refit:
            raise NotFittedError

        y_pred_train = clf.predict(x_train)

    except NotFittedError:

        if sample_weight is not None:
            clf.fit(x_train, y_train, sample_weight=sample_weight)
        else:
            clf.fit(x_train, y_train)

        y_pred_train = clf.predict(x_train)

    train_accuracy = accuracy_score(y_train, y_pred_train)

    return clf, y_pred_train, train_accuracy


def report(
    clf,
    x_train,
    y_train,
    x_test,
    y_test,
    sample_weight=None,
    refit=False,
    importance_plot=False,
    confusion_labels=None,
    feature_labels=None,
    verbose=True,
):
    """
    Trains passed model if not already trained and
    reports various metrics of the model.

    """

    dump = dict()

    # Train classifier if not already trained
    clf, train_predictions, train_accuracy = train_classifier(
        clf, x_train, y_train, sample_weight=sample_weight, refit=refit
    )

    # Test the model
    test_predictions = clf.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    y_probs = clf.predict_proba(x_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_probs)

    # Print the model
    print(clf)

    print("\n" + "=" * 30 + ">" + " TRAIN-TEST DETAILS " + "<" + "=" * 30)
    print()

    # Metrics
    print("Train Accuracy: ", train_accuracy)
    print("Test Accuracy: ", test_accuracy)

    print("-" * 50)

    print("Area Under ROC: ", roc_auc)

    print("-" * 50)

    print("\n" + "=" * 30 + ">" + " CLASSIFICATION REPORT " + "<" + "=" * 30)
    print()

    # Classification Report
    clf_report = classification_report(y_test, test_predictions, output_dict=True)

    print(
        classification_report(y_test, test_predictions, target_names=confusion_labels)
    )

    if verbose:

        print("\n" + "=" * 30 + ">" + " CONFUSION MATRIX " + "<" + "=" * 30)

        # Confusion Matrix Heatmap
        display(
            confusion_plot(
                confusion_matrix(y_test, test_predictions), labels=confusion_labels
            )
        )

        print("\n" + "=" * 30 + ">" + " PLOTS " + "<" + "=" * 30)

        # Feature importance plot
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
        roc_axes = axes[0, 0]
        pr_axes = axes[0, 1]
        importance = None

        if importance_plot:
            if not feature_labels:
                raise RuntimeError(
                    "feature_labels argument not passed when importance_plot is True"
                )

            try:
                importance = pd.Series(
                    clf.feature_importances_, index=feature_labels
                ).sort_values(ascending=False)

            except AttributeError:
                try:
                    importance = pd.Series(
                        clf.coef_.ravel(), index=feature_labels
                    ).sort_values(ascending=False)
                except AttributeError:
                    pass

            if importance is not None:
                grid_spec = axes[0, 0].get_gridspec()

                for ax in axes[:, 0]:
                    ax.remove()

                large_axs = fig.add_subplot(grid_spec[0:, 0])

                # Plot importance curve
                plot_feature_importance(
                    importance=importance.values,
                    feature_labels=importance.index,
                    ax=large_axs,
                )

                large_axs.axvline(x=0)

                roc_axes = axes[0, 1]
                pr_axes = axes[1, 1]
            else:
                for ax in axes[1, :]:
                    ax.remove()

        else:
            for ax in axes[1, :]:
                ax.remove()

        # ROC and Precision-Recall curves
        clf_name = clf.__class__.__name__
        plot_roc(y_test, y_probs, clf_name, ax=roc_axes)
        plot_precision_recall(y_test, y_probs, clf_name, ax=pr_axes)

        fig.subplots_adjust(wspace=5)
        fig.tight_layout()
        display(fig)

    # Dump to report dict
    dump = dict(
        clf=clf,
        train_acc=train_accuracy,
        train_predictions=train_predictions,
        test_acc=test_accuracy,
        test_predictions=test_predictions,
        test_probs=y_probs,
        report=clf_report,
        roc_auc=roc_auc,
    )

    return clf, dump


def compare_models(y_test=None, clf_reports=[], labels=[]):
    """
    Compare evaluation metrics for the True Positive class [1] of
    binary classifiers passed in the argument and plot ROC and PR curves.

    """

    ## Classifier Labels
    default_names = [rep["clf"].__class__.__name__ for rep in clf_reports]
    clf_names = labels if len(labels) == len(clf_reports) else default_names

    ## Compare Table
    table = dict()
    index = [
        "Train Accuracy",
        "Test Accuracy",
        "ROC Area",
        "Precision",
        "Recall",
        "F1-score",
    ]
    for i in range(len(clf_reports)):
        train_acc = round(clf_reports[i]["train_acc"], 3)
        test_acc = round(clf_reports[i]["test_acc"], 3)
        clf_probs = clf_reports[i]["test_probs"]
        roc_auc = clf_reports[i]["roc_auc"]

        # Get metrics of True Positive class from sklearn classification_report
        # Precision, Recall, F1-score, and the last one is Support, which I don't use
        true_positive_metrics = list(clf_reports[i]["report"]["1"].values())[:3]

        table[clf_names[i]] = [
            train_acc,
            test_acc,
            roc_auc,
        ] + true_positive_metrics

    table = pd.DataFrame(data=table, index=index)

    ## Compare Plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))

    # ROC and Precision-Recall
    for i in range(len(clf_reports)):
        clf_probs = clf_reports[i]["test_probs"]
        plot_roc(y_test, clf_probs, label=clf_names[i], compare=True, ax=axes[0])
        plot_precision_recall(
            y_test, clf_probs, label=clf_names[i], compare=True, ax=axes[1]
        )

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="green")

    fig.tight_layout()

    plt.close()

    return table.T, fig
