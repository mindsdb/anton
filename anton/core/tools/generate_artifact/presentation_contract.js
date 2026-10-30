/* Fail clearly when a presentation treats control state as a scalar or prints an object. */
(() => {
  window.CoworkSelection = selected => {
    const values = Object.freeze({...selected}), keys = Object.keys(values);
    const usage = keys.map(key => 'selection[' + JSON.stringify(key) + ']').join(', ') || 'an empty selection';
    return new Proxy(values, {get(target, key, receiver) {
      if (key === Symbol.toPrimitive) return () => {
        throw new TypeError('render receives a selection object. Read ' + usage + '; do not use the entire object as a string, dictionary key or value.');
      };
      if (typeof key === 'string' && !Reflect.has(target, key) && key !== 'toJSON') {
        throw new TypeError('Unknown control ' + JSON.stringify(key) + '. Available control values: ' + usage + '.');
      }
      return Reflect.get(target, key, receiver);
    }});
  };
  window.CoworkText = value => {
    if (typeof value === 'object' && value !== null || typeof value === 'function') {
      throw new TypeError('Presentation text needs a specific string or number. Select and format a source field instead of printing an object.');
    }
    if (typeof value === 'number' && !Number.isFinite(value)) {
      throw new TypeError('Presentation text contains a non-finite number. Check the calculation and retain unavailable values as null.');
    }
    return value;
  };
})();
