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
  GeneratorUnsupervisedActivations,
  UnsupervisedAnyActivationsRequest,
  UnsupervisedAnyActivationsResponse,
} from '../../datasets-page/unsupervised-activations/generator-unsupervised-activations';
import { Subscription } from 'rxjs';
import { LoadAnyActivationsImage } from '../../../directives/load-any-activations-image';
import { ZoomImage } from '../../../directives/zoom-image';
import {
  GenericUnsupervised,
  UnsupervisedCoordinate,
} from '../../shared/generic-unsupervised/generic-unsupervised';

@Component({
  selector: 'app-content-unsupervised-any-activations',
  imports: [
    LoadAnyActivationsImage,
    ZoomImage,
    GenericUnsupervised,
  ],
  templateUrl: './content-unsupervised-any-activations.html',
  styleUrl: './content-unsupervised-any-activations.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentUnsupervisedAnyActivations implements OnDestroy {
  model = input.required<string>();
  ref = input.required<string>();
  dataset = input.required<string>();
  label = input.required<string>();
  filename = input.required<string>();
  type = signal<string>('tsne');

  response = signal<UnsupervisedAnyActivationsResponse | null>(null);

  coordinates = computed<UnsupervisedCoordinate[]>(() => {
    const res = this.response();
    if (!res) return [];
    return res.label_filenames_coords;
  });

  private subscription: Subscription | null = null;

  unsupervisedActivationsService = inject(GeneratorUnsupervisedActivations);

  constructor() {
    effect(() => {
      if (this.subscription) {
        this.subscription.unsubscribe();
      }

      this.response.set(null);

      const request: UnsupervisedAnyActivationsRequest = {
        model: this.model(),
        ref: this.ref(),
        dataset: this.dataset(),
        label: this.label(),
        filename: this.filename(),
        type: this.type(),
      };

      this.subscription = this.unsupervisedActivationsService
        .unsupervisedAnyActivations(request)
        .subscribe((response: UnsupervisedAnyActivationsResponse) => {
          this.response.set(response);
        });
    });
  }

  ngOnDestroy(): void {
    if (this.subscription) {
      this.subscription.unsubscribe();
    }
  }
}
