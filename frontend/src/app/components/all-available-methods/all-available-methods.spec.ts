import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AllAvailableMethods } from './all-available-methods';

describe('AllAvailableMethods', () => {
  let component: AllAvailableMethods;
  let fixture: ComponentFixture<AllAvailableMethods>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AllAvailableMethods]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AllAvailableMethods);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
