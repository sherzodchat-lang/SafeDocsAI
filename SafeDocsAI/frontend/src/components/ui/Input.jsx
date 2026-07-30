import React from 'react';
import { cn } from '../../lib/utils';

const Input = React.forwardRef(({ className, type, ...props }, ref) => {
    return (
        <input
            type={type}
            className={cn(
                'flex h-10 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 shadow-sm transition-colors placeholder:text-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/30 focus-visible:border-[#1f3a60] disabled:cursor-not-allowed disabled:opacity-50',
                className,
            )}
            ref={ref}
            {...props}
        />
    );
});

Input.displayName = 'Input';

export default Input;
