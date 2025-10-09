import { Pipe, PipeTransform, inject } from '@angular/core';
import { ConstantStore } from '../stores/constants/constants.store';

@Pipe({
  name: 'translateXaiMethods',
  standalone: true,
})
export class TranslateXaiMethodsPipe implements PipeTransform {
  private readonly constantStore = inject(ConstantStore);

  transform(value: unknown): unknown {
    const translations = this.constantStore.translations();
    return translations[value as string] || '';
  }
}
