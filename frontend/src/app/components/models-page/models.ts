import { ChangeDetectionStrategy, Component, signal } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { GenericModelSelector } from '../shared/generic-model-selector/generic-model-selector';
import { ModelLayersResponse } from '../../interfaces/model-layers';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-models',
  imports: [GenericModelSelector],
  templateUrl: './models.html',
  styleUrl: './models.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Models {
  currentModel = signal<string | null>(null);

  layers = httpResource<ModelLayersResponse>(() => {
    const model = this.currentModel();
    if (!model) return undefined;
    return {
      url: `${environment.apiUrl}/model-layers`,
      params: { model },
    };
  });
}
