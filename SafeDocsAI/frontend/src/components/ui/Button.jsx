import React from 'react';
import { cn } from '../../lib/utils';
import { Loader2 } from 'lucide-react';

const Button = React.forwardRef(({
    className,
    variant = 'primary',
    size = 'default',
    isLoading = false,
    disabled = false,
    children,
    ...props
}, ref) => {
    const variants = {
        primary: 'bg-[#1f3a60] text-white hover:bg-[#162945] shadow-[0_8px_18px_rgba(31,58,96,0.22)]',
        secondary: 'bg-[#c5a059] text-white hover:bg-[#b18f4e] shadow-[0_8px_18px_rgba(197,160,89,0.28)]',
        outline: 'border border-slate-300 bg-white text-slate-700 hover:bg-slate-50',
        ghost: 'text-slate-600 hover:bg-slate-100 hover:text-slate-900',
        destructive: 'bg-red-600 text-white hover:bg-red-700',
    };

    const sizes = {
        default: 'h-10 px-4 py-2 text-sm',
        sm: 'h-8 px-3 text-xs',
        lg: 'h-12 px-6 text-base',
        icon: 'h-10 w-10',
    };

    return (
        <button
            ref={ref}
            className={cn(
                'inline-flex items-center justify-center gap-2 rounded-lg font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/40 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-60',
                variants[variant],
                sizes[size],
                className,
            )}
            disabled={isLoading || disabled}
            {...props}
        >
            {isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
            {children}
        </button>
    );
});

Button.displayName = 'Button';

export { Button };
