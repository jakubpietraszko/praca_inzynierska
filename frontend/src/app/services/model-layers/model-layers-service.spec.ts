import { TestBed } from '@angular/core/testing';

import { ModelLayersService } from './model-layers-service';

describe('ModelLayersService', () => {
  let service: ModelLayersService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ModelLayersService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
