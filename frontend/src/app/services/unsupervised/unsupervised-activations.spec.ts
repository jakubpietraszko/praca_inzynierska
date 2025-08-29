import { TestBed } from '@angular/core/testing';

import { UnsupervisedActivationsService } from './unsupervised-activations';

describe('UnsupervisedActivationsService', () => {
  let service: UnsupervisedActivationsService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(UnsupervisedActivationsService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
