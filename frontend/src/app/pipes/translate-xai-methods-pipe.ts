import { Pipe, PipeTransform } from '@angular/core';
import { constants } from '../../constants';

@Pipe({
  name: 'translateXaiMethods',
})
export class TranslateXaiMethodsPipe implements PipeTransform {
  transform(value: unknown, ...args: unknown[]): unknown {
    return (
      constants.heatmaps[value as keyof typeof constants.heatmaps]
        ?.translation || ''
    );
  }
}
