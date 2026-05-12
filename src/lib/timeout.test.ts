import { test, describe, mock } from 'node:test';
import assert from 'node:assert';
import { timeoutPromise } from './timeout.ts';

describe('timeoutPromise', () => {
  test('resolves when the promise completes before timeout', async () => {
    const data = { success: true };
    const promise = Promise.resolve(data);
    const result = await timeoutPromise(promise, 100);
    assert.deepStrictEqual(result, data);
  });

  test('rejects when the promise takes longer than timeout', async () => {
    mock.timers.enable();
    const slowPromise = new Promise((resolve) => setTimeout(() => resolve('done'), 200));

    const timeoutedPromise = timeoutPromise(slowPromise, 100);

    mock.timers.tick(150);

    await assert.rejects(timeoutedPromise, {
      message: 'Request timeout'
    });

    mock.timers.reset();
  });
});
