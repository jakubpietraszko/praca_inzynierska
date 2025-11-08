import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { ConfusionMatrix } from '../../../../generated-interfaces/model_dataset_statistics';
import { MatCardModule } from '@angular/material/card';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-confusion-matrix',
  imports: [MatCardModule, CommonModule],
  templateUrl: './confusion-matrix.html',
  styleUrl: './confusion-matrix.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ConfusionMatrixComponent {
  confusionMatrix = input.required<ConfusionMatrix>();

  classNames = computed(() => {
    const matrix = this.confusionMatrix();
    if (!matrix) return [];
    return Object.keys(matrix);
  });

  maxConfusionValue = computed(() => {
    const matrix = this.confusionMatrix();
    if (!matrix) return 0;

    let max = 0;
    Object.values(matrix).forEach((row) => {
      Object.values(row).forEach((value) => {
        max = Math.max(max, value);
      });
    });
    return max;
  });

  getConfusionValue(actualClass: string, predictedClass: string): number {
    const matrix = this.confusionMatrix();
    if (!matrix) return 0;
    return matrix[actualClass]?.[predictedClass] || 0;
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
