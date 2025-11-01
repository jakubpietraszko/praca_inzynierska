import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
  OnDestroy,
  OnInit,
  signal,
} from '@angular/core';
import {
  GeneratorUnsupervisedActivations,
  UnsupervisedActivationsRequest,
  UnsupervisedActivationsResponse,
} from '../generator-unsupervised-activations';
import { Subscription } from 'rxjs';
import { LoadActivationImage } from '../../../../directives/load-activation-image';
import {
  GenericUnsupervised,
  UnsupervisedCoordinate,
} from '../../../shared/generic-unsupervised/generic-unsupervised';
import { ZoomImage } from '../../../../directives/zoom-image';

@Component({
  selector: 'app-content-unsupervised-activations',
  imports: [LoadActivationImage, GenericUnsupervised, ZoomImage],
  templateUrl: './content-unsupervised-activations.html',
  styleUrl: './content-unsupervised-activations.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentUnsupervisedActivations implements OnInit, OnDestroy {
  model = input.required<string>();
  dataset = input.required<string>();
  label = input.required<string>();
  filename = input.required<string>();
  type = signal<string>('tsne');

  response = signal<UnsupervisedActivationsResponse | null>(null);

  coordinates = computed<UnsupervisedCoordinate[]>(() => {
    const res = this.response();
    if (!res) return [];
    return res.label_filenames_coords;
  });

  private subscription = new Subscription();

  unsupervisedActivationsService = inject(GeneratorUnsupervisedActivations);

  ngOnInit(): void {
    const request: UnsupervisedActivationsRequest = {
      model: this.model(),
      dataset: this.dataset(),
      label: this.label(),
      filename: this.filename(),
      type: this.type(),
    };

    this.subscription.add(
      this.unsupervisedActivationsService
        .unsupervisedActivations(request)
        .subscribe((response: UnsupervisedActivationsResponse) => {
          this.response.set(response);
        }),
    );
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }
}
