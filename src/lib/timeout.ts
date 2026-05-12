/**
 * Utility to wrap a promise with a timeout.
 * If the promise does not resolve within the specified time, it rejects.
 */
export function timeoutPromise<T>(promise: Promise<T>, ms: number): Promise<T> {
  const timeout = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error("Request timeout")), ms)
  );

  return Promise.race([promise, timeout]);
}
