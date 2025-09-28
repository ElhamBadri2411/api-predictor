"""
Simplified ML Model Trainer: Logistic Regression Only

Trains only Logistic Regression since it's the chosen model for production.
Optimized for speed and simplicity without the Random Forest comparison overhead.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import numpy as np
import time
import os
from ml.features import FeatureExtractor
from ml.data_generator import TrainingDataGenerator


class SimpleLRTrainer:
    """Train Logistic Regression model for API prediction"""

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.model = None
        self.results = {}

    def train(self, n_samples: int = 25000, test_size: float = 0.2):
        """Train Logistic Regression model"""

        print(f"🚀 Training Logistic Regression with {n_samples:,} samples...")

        # Prepare data
        X, y = self._prepare_data(n_samples)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"📊 Training: {X_train.shape}, Test: {X_test.shape}")
        print(f"📊 Feature stats: mean={X.mean():.3f}, std={X.std():.3f}")

        # Train model
        print(f"\n🔧 Training Logistic Regression...")

        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            C=1.0  # L2 regularization
        )

        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Evaluate model
        self.results = self._evaluate_model(X_train, X_test, y_train, y_test, training_time)

        # Analysis
        self._analyze_results(y_test)

        return self.results

    def _prepare_data(self, n_samples: int):
        """Generate training data and extract features"""

        print("📊 Generating training data...")
        generator = TrainingDataGenerator()
        samples = generator.generate_training_samples(n_samples)

        positive = sum(1 for s in samples if s['label'] == 1)
        print(f"Generated {len(samples)} samples: {positive} positive, {len(samples)-positive} negative")

        print("🔧 Extracting features...")
        X, y = [], []

        for i, sample in enumerate(samples):
            if i % 2500 == 0 and i > 0:
                print(f"  Processed {i}/{len(samples)} samples")

            try:
                features = self.extractor.extract(
                    sample['history'],
                    sample['candidate'],
                    sample.get('prompt')
                )
                X.append(features)
                y.append(sample['label'])
            except Exception as e:
                print(f"  Warning: Skipped sample {i}: {e}")
                continue

        return np.array(X), np.array(y)

    def _evaluate_model(self, X_train, X_test, y_train, y_test, training_time):
        """Thoroughly evaluate the model"""

        print(f"  📊 Evaluating model...")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Core metrics
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        # Speed test - critical for API performance
        print(f"  ⚡ Speed testing...")
        inference_times = []
        single_sample = X_test[0:1]

        # Warm up
        for _ in range(10):
            self.model.predict(single_sample)

        # Actual speed test
        for _ in range(200):
            start = time.time()
            self.model.predict(single_sample)
            inference_times.append((time.time() - start) * 1000)  # ms

        avg_inference_ms = np.mean(inference_times)
        p95_inference_ms = np.percentile(inference_times, 95)

        # Model complexity
        import pickle
        model_size_mb = len(pickle.dumps(self.model)) / 1024 / 1024

        # Cross-validation for robustness
        print(f"  📊 Cross-validation...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='roc_auc')

        results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'auc_score': auc_score,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'training_time_s': training_time,
            'avg_inference_ms': avg_inference_ms,
            'p95_inference_ms': p95_inference_ms,
            'model_size_mb': model_size_mb,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        # Print results
        print(f"\n✅ Logistic Regression Results:")
        print(f"   Training Accuracy: {train_acc:.3f}")
        print(f"   Test Accuracy: {test_acc:.3f}")
        print(f"   AUC Score: {auc_score:.3f}")
        print(f"   CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"   Training Time: {training_time:.1f}s")
        print(f"   Avg Inference: {avg_inference_ms:.2f}ms")
        print(f"   P95 Inference: {p95_inference_ms:.2f}ms")
        print(f"   Model Size: {model_size_mb:.3f}MB")

        return results

    def _analyze_results(self, y_test):
        """Analyze results and feature importance"""

        print(f"\n{'='*50}")
        print(f"📊 DETAILED ANALYSIS")
        print(f"{'='*50}")

        # Confusion matrix
        cm = confusion_matrix(y_test, self.results['predictions'])
        tn, fp, fn, tp = cm.ravel()

        print(f"\n📊 Confusion Matrix:")
        print(f"   True Neg: {tn:4d}  False Pos: {fp:4d}")
        print(f"   False Neg: {fn:4d}  True Pos: {tp:4d}")
        print(f"   Precision: {tp/(tp+fp):.3f}  Recall: {tp/(tp+fn):.3f}")

        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, self.results['predictions']))

        # Feature importance (coefficients)
        self._analyze_feature_importance()

        # Speed requirement check
        print(f"\n⚡ SPEED ANALYSIS:")
        print(f"Average inference: {self.results['avg_inference_ms']:.2f}ms")
        print(f"P95 inference: {self.results['p95_inference_ms']:.2f}ms")

        if self.results['p95_inference_ms'] < 1.0:
            print("✅ Meets strict speed requirement (P95 < 1ms)")
        elif self.results['avg_inference_ms'] < 1.0:
            print("✅ Meets average speed requirement (avg < 1ms)")
        else:
            print("⚠️ May struggle with speed requirement")

    def _analyze_feature_importance(self):
        """Analyze feature importance from LR coefficients"""

        print(f"\n🔍 FEATURE IMPORTANCE (Top 10 by |coefficient|):")
        print("-" * 50)

        feature_names = self._get_feature_names()
        coefficients = self.model.coef_[0]

        # Get absolute coefficients for importance ranking
        abs_coef = np.abs(coefficients)
        indices = np.argsort(abs_coef)[::-1]

        for i in range(min(10, len(indices))):
            idx = indices[i]
            coef_val = coefficients[idx]
            abs_val = abs_coef[idx]
            direction = "+" if coef_val > 0 else "-"

            print(f"{i+1:2d}. {feature_names[idx]:<30} {direction}{abs_val:.4f}")

    def _get_feature_names(self):
        """Get human-readable feature names"""
        names = []

        # Sequence features (0-9)
        for method in self.extractor.methods:
            names.append(f"seq_method_{method.lower()}")
        names.extend([
            "seq_resource_stickiness", "seq_depth_progression",
            "seq_loop_detection", "seq_read_write_ratio", "seq_velocity"
        ])

        # Transition features (10-34)
        for from_m in self.extractor.methods:
            for to_m in self.extractor.methods:
                names.append(f"trans_{from_m.lower()}_to_{to_m.lower()}")

        # Resource features (35-39)
        names.extend([
            "res_same_resource", "res_similarity", "res_depth_change",
            "res_has_id", "res_in_recent"
        ])

        # Prompt features (40-45)
        names.extend([
            "prompt_has_prompt", "prompt_verb_alignment", "prompt_resource_mentioned",
            "prompt_complexity", "prompt_has_action_words", "prompt_unsafe"
        ])

        # Pattern features (46-50)
        names.extend([
            "pattern_list_to_detail", "pattern_detail_to_update", "pattern_create_to_view",
            "pattern_update_to_view", "pattern_safe_operation"
        ])

        return names

    def save_model(self, model_path: str = 'data/model.pkl',
                   extractor_path: str = 'data/feature_extractor.pkl'):
        """Save the trained model"""

        if not self.model:
            raise ValueError("No model trained yet. Run train() first.")

        os.makedirs('data', exist_ok=True)

        # Save model and extractor
        joblib.dump(self.model, model_path)
        joblib.dump(self.extractor, extractor_path)

        # Save results summary
        import json
        summary = {
            'model_type': 'Logistic Regression',
            'test_accuracy': float(self.results['test_accuracy']),
            'auc_score': float(self.results['auc_score']),
            'avg_inference_ms': float(self.results['avg_inference_ms']),
            'model_size_mb': float(self.results['model_size_mb']),
            'meets_speed_requirement': bool(self.results['p95_inference_ms'] < 1.0)
        }

        with open('data/model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Model saved successfully:")
        print(f"   Model: {model_path}")
        print(f"   Extractor: {extractor_path}")
        print(f"   Summary: data/model_summary.json")


def train_production_model(n_samples: int = 25000):
    """Train production-ready Logistic Regression model"""

    trainer = SimpleLRTrainer()

    # Train model
    results = trainer.train(n_samples=n_samples)

    # Save for production use
    trainer.save_model()

    print(f"\n🎉 Training complete!")
    print(f"✅ Model ready for production use")
    print(f"📊 Final accuracy: {results['test_accuracy']:.1%}")
    print(f"⚡ Inference speed: {results['avg_inference_ms']:.2f}ms")

    return trainer


def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Train Logistic Regression model for API prediction')
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=25000,
        help='Number of training samples to generate (default: 25000)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data',
        help='Output directory for saved model (default: data)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick training with 5000 samples'
    )

    args = parser.parse_args()

    # Handle quick mode
    if args.quick:
        n_samples = 5000
        print("🏃 Quick training mode (5K samples)")
    else:
        n_samples = args.samples

    print(f"🚀 Starting training with {n_samples:,} samples...")
    print(f"📁 Output directory: {args.output_dir}")

    # Create trainer
    trainer = SimpleLRTrainer()

    # Train model
    results = trainer.train(n_samples=n_samples, test_size=args.test_size)

    # Save model to specified directory
    model_path = f"{args.output_dir}/model.pkl"
    extractor_path = f"{args.output_dir}/feature_extractor.pkl"
    trainer.save_model(model_path, extractor_path)

    print(f"\n🎉 Training complete!")
    print(f"✅ Model ready for production use")
    print(f"📊 Final accuracy: {results['test_accuracy']:.1%}")
    print(f"⚡ Inference speed: {results['avg_inference_ms']:.2f}ms")
    print(f"💾 Model saved to: {model_path}")

    return trainer


if __name__ == "__main__":
    main()