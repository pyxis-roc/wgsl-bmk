abstract class Holder<T> {
	/**
	 * Create a new holder with the given initial value.
	 * @param value Initial value of the holder.
	 */
	constructor(public value: T) {}
}

export class NumberHolder extends Holder<number> {}
export class BooleanHolder extends Holder<boolean> {
	/**
	 * Set the value to true if it is currently false, and return the previous value.
	 * @returns The previous value in the holder.
	 */
	public testAndSet(): boolean {
		if (this.value) {
			return true;
		}
		this.value = true;
		return false;
	}
}
