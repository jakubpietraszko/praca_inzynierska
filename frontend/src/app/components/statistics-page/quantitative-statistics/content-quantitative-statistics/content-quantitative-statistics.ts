import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  input,
  OnDestroy,
  signal,
} from '@angular/core';
import {
  GeneratorStatistics,
  QuantitativeStatisticsRequest,
  QuantitativeStatisticsResponse,
} from '../../generator-statistics';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatCardModule } from '@angular/material/card';
import { MatTableModule } from '@angular/material/table';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-content-quantitative-statistics',
  imports: [
    MatProgressSpinnerModule,
    MatCardModule,
    MatTableModule,
    CommonModule,
  ],
  templateUrl: './content-quantitative-statistics.html',
  styleUrl: './content-quantitative-statistics.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentQuantitativeStatistics implements OnDestroy {
  private readonly statisticsService = inject(GeneratorStatistics);

  private subscription = new Subscription();

  model = input.required<string>();
  dataset = input.required<string>();

  quantitativeStatisticsResponse =
    signal<QuantitativeStatisticsResponse | null>(null);

  displayedColumns: string[] = [
    'class',
    'precision',
    'recall',
    'f1',
    'support',
  ];

  constructor() {
    effect(() => {
      const request: QuantitativeStatisticsRequest = {
        dataset: this.dataset(),
        model: this.model(),
      };

      if (this.subscription) {
        this.subscription.unsubscribe();
      }

      this.quantitativeStatisticsResponse.set(null);

      this.subscription = new Subscription();

      this.subscription.add(
        this.statisticsService
          .quantitativeStatistics(request)
          .subscribe((response: QuantitativeStatisticsResponse) => {
            this.quantitativeStatisticsResponse.set(response);
          })
      );
    });
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }

  classNames = computed(() => {
    const response = this.quantitativeStatisticsResponse();
    if (!response?.confusion_dict) return [];
    return Object.keys(response.confusion_dict);
  });

  allCorrect = computed(() => {
    const response = this.quantitativeStatisticsResponse();
    if (!response?.confusion_dict) return 0;

    let correct = 0;
    Object.keys(response.confusion_dict).forEach((actualClass) => {
      correct += response.confusion_dict[actualClass]?.[actualClass] || 0;
    });
    return correct;
  });

  allTotal = computed(() => {
    const response = this.quantitativeStatisticsResponse();
    if (!response?.confusion_dict) return 0;

    let total = 0;
    Object.keys(response.confusion_dict).forEach((actualClass) => {
      Object.keys(response.confusion_dict[actualClass]).forEach(
        (predictedClass) => {
          total += response.confusion_dict[actualClass][predictedClass];
        }
      );
    });
    return total;
  });

  macroAccuracy = computed(() => {
    const total = this.allTotal();
    const correct = this.allCorrect();
    return total > 0 ? correct / total : 0;
  });

  microTruePositives = computed(() => {
    const response = this.quantitativeStatisticsResponse();
    const classNames = this.classNames();
    const result = new Map<string, number>();
    if (!response?.confusion_dict || classNames.length === 0) return result;

    classNames.forEach((className) => {
      const truePositive = response.confusion_dict[className]?.[className] || 0;
      result.set(className, truePositive);
    });

    return result;
  });

  microFalseNegatives = computed(() => {
    const response = this.quantitativeStatisticsResponse();
    const classNames = this.classNames();
    const result = new Map<string, number>();
    if (!response?.confusion_dict || classNames.length === 0) return result;

    classNames.forEach((className) => {
      let falseNegative = 0;

      Object.keys(response.confusion_dict[className] || {}).forEach(
        (predictedClass) => {
          if (predictedClass !== className) {
            falseNegative += response.confusion_dict[className][predictedClass];
          }
        }
      );

      result.set(className, falseNegative);
    });

    return result;
  });

  microFalsePositives = computed(() => {
    const response = this.quantitativeStatisticsResponse();
    const classNames = this.classNames();
    const result = new Map<string, number>();
    if (!response?.confusion_dict || classNames.length === 0) return result;

    classNames.forEach((className) => {
      let falsePositive = 0;

      classNames.forEach((actualClass) => {
        if (actualClass !== className) {
          falsePositive +=
            response.confusion_dict[actualClass]?.[className] || 0;
        }
      });

      result.set(className, falsePositive);
    });

    return result;
  });

  microPrecision = computed(() => {
    const truePositives = this.microTruePositives();
    const falsePositives = this.microFalsePositives();
    const result = new Map<string, number>();

    truePositives.forEach((tp, className) => {
      const fp = falsePositives.get(className) || 0;
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      result.set(className, precision);
    });

    return result;
  });

  microRecall = computed(() => {
    const truePositives = this.microTruePositives();
    const falseNegatives = this.microFalseNegatives();
    const result = new Map<string, number>();

    truePositives.forEach((tp, className) => {
      const fn = falseNegatives.get(className) || 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      result.set(className, recall);
    });

    return result;
  });

  microF1 = computed(() => {
    const precisions = this.microPrecision();
    const recalls = this.microRecall();
    const result = new Map<string, number>();

    precisions.forEach((precision, className) => {
      const recall = recalls.get(className) || 0;
      const f1 =
        precision + recall > 0
          ? (2 * precision * recall) / (precision + recall)
          : 0;
      result.set(className, f1);
    });

    return result;
  });

  microSupport = computed(() => {
    const response = this.quantitativeStatisticsResponse();
    const classNames = this.classNames();
    const result = new Map<string, number>();
    if (!response?.confusion_dict || classNames.length === 0) return result;

    classNames.forEach((className) => {
      let support = 0;

      Object.keys(response.confusion_dict[className] || {}).forEach(
        (predictedClass) => {
          support += response.confusion_dict[className][predictedClass];
        }
      );

      result.set(className, support);
    });

    return result;
  });

  macroPrecision = computed(() => {
    const precisions = this.microPrecision();
    let sum = 0;
    precisions.forEach((precision: number) => {
      sum += precision;
    });
    return precisions.size > 0 ? sum / precisions.size : 0;
  });

  macroRecall = computed(() => {
    const recalls = this.microRecall();
    let sum = 0;
    recalls.forEach((recall: number) => {
      sum += recall;
    });
    return recalls.size > 0 ? sum / recalls.size : 0;
  });

  macroF1 = computed(() => {
    const f1s = this.microF1();
    let sum = 0;
    f1s.forEach((f1) => {
      sum += f1;
    });
    return f1s.size > 0 ? sum / f1s.size : 0;
  });

  macroWeightedPrecision = computed(() => {
    const precisions = this.microPrecision();
    const supports = this.microSupport();
    let weightedSum = 0;
    let totalSupport = 0;

    precisions.forEach((precision: number, className: string) => {
      const support = supports.get(className) || 0;
      weightedSum += precision * support;
      totalSupport += support;
    });

    return totalSupport > 0 ? weightedSum / totalSupport : 0;
  });

  macroWeightedRecall = computed(() => {
    const recalls = this.microRecall();
    const supports = this.microSupport();
    let weightedSum = 0;
    let totalSupport = 0;

    recalls.forEach((recall: number, className: string) => {
      const support = supports.get(className) || 0;
      weightedSum += recall * support;
      totalSupport += support;
    });

    return totalSupport > 0 ? weightedSum / totalSupport : 0;
  });

  macroWeightedF1 = computed(() => {
    const f1s = this.microF1();
    const supports = this.microSupport();
    let weightedSum = 0;
    let totalSupport = 0;

    f1s.forEach((f1, className) => {
      const support = supports.get(className) || 0;
      weightedSum += f1 * support;
      totalSupport += support;
    });

    return totalSupport > 0 ? weightedSum / totalSupport : 0;
  });

  accuracy = computed(() => {
    return this.macroAccuracy();
  });

  microAccuracy = computed(() => {
    const response = this.quantitativeStatisticsResponse();
    const classNames = this.classNames();
    const result = new Map<string, number>();
    if (!response?.confusion_dict || classNames.length === 0) return result;

    const totalSamples = this.allTotal();

    classNames.forEach((className) => {
      const truePositive = response.confusion_dict[className]?.[className] || 0;
      let trueNegative = 0;

      classNames.forEach((otherClass) => {
        if (otherClass !== className) {
          trueNegative +=
            response.confusion_dict[otherClass]?.[otherClass] || 0;
        }
      });

      let falsePositive = 0;
      classNames.forEach((actualClass) => {
        if (actualClass !== className) {
          falsePositive +=
            response.confusion_dict[actualClass]?.[className] || 0;
        }
      });

      let falseNegative = 0;
      Object.keys(response.confusion_dict[className] || {}).forEach(
        (predictedClass) => {
          if (predictedClass !== className) {
            falseNegative += response.confusion_dict[className][predictedClass];
          }
        }
      );

      const accuracy =
        totalSamples > 0 ? (truePositive + trueNegative) / totalSamples : 0;
      result.set(className, accuracy);
    });

    return result;
  });

  classMetrics = computed(() => {
    const classNames = this.classNames();
    const precisions = this.microPrecision();
    const recalls = this.microRecall();
    const f1s = this.microF1();
    const supports = this.microSupport();
    const accuracies = this.microAccuracy();

    return classNames.map((className: string) => ({
      class: className,
      precision: precisions.get(className) || 0,
      recall: recalls.get(className) || 0,
      f1: f1s.get(className) || 0,
      accuracy: accuracies.get(className) || 0,
      support: supports.get(className) || 0,
    }));
  });

  maxConfusionValue = computed(() => {
    const response = this.quantitativeStatisticsResponse();
    if (!response?.confusion_dict) return 0;

    let max = 0;
    Object.values(response.confusion_dict).forEach((row) => {
      Object.values(row).forEach((value) => {
        max = Math.max(max, value);
      });
    });
    return max;
  });

  getConfusionValue(actualClass: string, predictedClass: string): number {
    const response = this.quantitativeStatisticsResponse();
    if (!response?.confusion_dict) return 0;
    return response.confusion_dict[actualClass]?.[predictedClass] || 0;
  }

  getCellColor(actualClass: string, predictedClass: string): string {
    const value = this.getConfusionValue(actualClass, predictedClass);
    const maxValue = this.maxConfusionValue();

    if (maxValue === 0) return '#f5f5f5';

    const intensity = value / maxValue;

    if (actualClass === predictedClass) {
      const greenIntensity = Math.floor(255 - intensity * 200);
      return `rgb(${greenIntensity}, 255, ${greenIntensity})`;
    } else {
      const redIntensity = Math.floor(255 - intensity * 200);
      return `rgb(255, ${redIntensity}, ${redIntensity})`;
    }
  }
}
